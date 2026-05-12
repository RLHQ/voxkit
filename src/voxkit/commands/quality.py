"""``voxkit quality`` 子命令入口。

扫描 ``<workdir>`` 内已存在的字幕 artifact（cues / proofread / translate），输出
``quality.report.json``。纯计算，零 LLM 零网络，因此不需要 provider/api key。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxkit.core.quality_metrics import build_quality_report, write_quality_report


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "quality",
        help=(
            "聚合字幕物理指标 + 风险分布，输出 quality.report.json。"
            "无 LLM/网络依赖，可对任意已生成 artifact 子集运行"
        ),
    )
    p.add_argument(
        "workdir",
        help="voxkit transcribe/proofread/translate 的 workdir",
    )
    p.add_argument(
        "--output",
        default=None,
        help="覆盖输出路径（默认 <workdir>/quality.report.json）",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    output = Path(args.output) if args.output else workdir / "quality.report.json"

    try:
        report = build_quality_report(workdir)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    write_quality_report(output, report)

    translations = sorted(report.translations.keys())
    has_proofread = "Y" if report.proofread_metrics is not None else "N"
    cue_count = report.cues_metrics.cue_count if report.cues_metrics else 0
    sys.stdout.write(
        f"quality report written: {output} "
        f"(cues={cue_count}, proofread={has_proofread}, "
        f"translations={translations})\n"
    )
    return 0
