"""``voxkit eval`` 子命令入口。

默认对照人类金标 SRT 字幕，评估 ``<workdir>`` 内 voxkit 输出的 reseg 质量。
零 LLM 零网络，纯计算——可在 CI 频繁跑。

加 ``--llm`` 时启用 L3 多维评分（语义 / 术语 / 切分 / 标点 / 节奏），通过
LLM 提供 boundary metrics 看不到的字幕质量维度。需要 LLM API key，慢且
有成本，**不进 CI**，release / PR review 时跑。

指标语义详见 ``docs/eval.md``、``docs/eval-baseline-observations.md``、
``docs/eval-methodology.md``。
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
            "默认零 LLM 零网络；--llm 开启 L3 多维评分"
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
        help="边界对齐容忍度（秒，默认 0.3，仅 boundary metrics 用）",
    )
    p.add_argument(
        "--output",
        default=None,
        help="覆盖 boundary 报告输出路径（默认 <workdir>/eval.report.json）",
    )
    # L3 LLM eval
    p.add_argument(
        "--llm",
        action="store_true",
        help="启用 LLM 多维评分（5 维 + 整体），输出 eval-llm.report.json",
    )
    p.add_argument(
        "--provider",
        default="deepseek",
        help="LLM provider（默认 deepseek），仅 --llm 时用",
    )
    p.add_argument(
        "--model",
        default=None,
        help="覆盖 provider 默认 model，仅 --llm 时用",
    )
    p.add_argument(
        "--max-groups-per-batch",
        type=int,
        default=10,
        help="单 LLM 请求评分的对齐组数（默认 10），仅 --llm 时用",
    )
    p.add_argument(
        "--llm-output",
        default=None,
        help="覆盖 LLM 报告输出路径（默认 <workdir>/eval-llm.report.json）",
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

    # ── L1 boundary eval（始终跑）────────────────────────────────────
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

    # ── L3 LLM eval（可选）──────────────────────────────────────────
    if not args.llm:
        return 0

    # 延迟导入：避免无 --llm 时拖入 LLM client（保持 default zero-LLM 契约）
    from voxkit.core.llm_eval import (
        build_llm_eval_report,
        write_llm_eval_report,
    )
    from voxkit.llm.client import LLMClient
    from voxkit.llm.providers import get_provider

    llm_output = (
        Path(args.llm_output) if args.llm_output else workdir / "eval-llm.report.json"
    )
    model_name = args.model or get_provider(args.provider).default_model

    try:
        with LLMClient(args.provider, model=model_name) as client:
            llm_report = build_llm_eval_report(
                workdir=workdir,
                reference=reference,
                language=args.lang,
                client=client,
                provider=args.provider,
                model=model_name,
                max_groups_per_batch=args.max_groups_per_batch,
            )
    except Exception as e:  # noqa: BLE001 — surface LLM/network failures cleanly
        sys.stderr.write(f"error: LLM eval failed: {e}\n")
        return 2

    write_llm_eval_report(llm_output, llm_report)

    agg = llm_report.scores_aggregate
    al = llm_report.alignment
    sys.stdout.write(
        f"\nllm eval report written: {llm_output}\n"
        f"  provider={llm_report.provider} model={llm_report.model} "
        f"promptHash={llm_report.promptHash[:8]}\n"
        f"  groups: total={al['groups']} both={al['groups_with_both']} "
        f"vk_only={al['groups_vk_only']} gold_only={al['groups_gold_only']}\n"
        f"  scores (mean / p10):\n"
        f"    semantic     {agg['semantic']['mean']:.2f} / {agg['semantic']['p10']:.0f}\n"
        f"    terminology  {agg['terminology']['mean']:.2f} / {agg['terminology']['p10']:.0f}\n"
        f"    segmentation {agg['segmentation']['mean']:.2f} / {agg['segmentation']['p10']:.0f}\n"
        f"    punctuation  {agg['punctuation']['mean']:.2f} / {agg['punctuation']['p10']:.0f}\n"
        f"    readability  {agg['readability']['mean']:.2f} / {agg['readability']['p10']:.0f}\n"
        f"    overall      {agg['overall']['mean']:.2f} / {agg['overall']['p10']:.1f}\n"
        f"  high_risk_groups: {len(llm_report.high_risk_groups)}\n"
        f"  tokens: prompt={llm_report.tokens['prompt']} "
        f"completion={llm_report.tokens['completion']}\n"
    )
    return 0
