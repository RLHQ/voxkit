"""``voxkit translate`` 子命令入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from voxkit.core.pricing import estimate_cost, format_cost, lookup_rates
from voxkit.core.translate_pipeline import TranslateRequest, run_translate
from voxkit.llm.errors import LLMError


def add_subparser(sub: argparse._SubParsersAction) -> None:
    from voxkit.llm.providers import PROVIDERS as _LLM_PROVIDERS

    provider_lines = "\n".join(
        f"    {name}: env={spec.api_key_env}, default_model={spec.default_model}"
        for name, spec in sorted(_LLM_PROVIDERS.items())
    )
    epilog = (
        "支持的 LLM provider（OpenAI-compatible 协议统一调用，credential 走 env）：\n"
        f"{provider_lines}\n\n"
        "--force 三档对应表（gate_force_overwrite 拒覆盖逻辑）：\n"
        "    artifact state    使用的 flag                  说明\n"
        "    (不存在)          (任意)                       直通\n"
        "    draft             --force                      覆盖 draft\n"
        "    reviewed          --force-reviewed             覆盖 reviewed（隐含 --force）\n"
        "    final             --force-final                覆盖 final（销毁人工 lock）\n\n"
        "想换 --speaker-prefix 但不重花 LLM token：加 --render-only。\n"
    )
    p = sub.add_parser(
        "translate",
        help=(
            "把字幕翻译到目标语言。输入优先 subtitles.proofread.json，缺失回落 "
            "subtitles.cues.json。需要 DEEPSEEK_API_KEY 或对应 provider key"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("workdir", help="voxkit transcribe 的 workdir")
    p.add_argument("--target-language", required=True, help="目标语言代码，例如 en / zh / ja")
    p.add_argument(
        "--source-language",
        default=None,
        help="覆盖源语言；缺省从输入 artifact 推断",
    )
    p.add_argument(
        "--style",
        choices=["literal", "natural", "subtitle", "technical"],
        default="subtitle",
        help="翻译风格（默认 subtitle，专为屏幕阅读优化）",
    )
    p.add_argument(
        "--length-policy",
        choices=["preserve", "subtitle-fit"],
        default="preserve",
        help="长度策略（默认 preserve；subtitle-fit 会要求模型主动压缩长度）",
    )
    p.add_argument(
        "--provider",
        default="deepseek",
        help=(
            "LLM provider 名（默认 deepseek）。"
            "当前注册：" + ", ".join(sorted(_LLM_PROVIDERS.keys())) +
            "；具体 env var 见 --help epilog"
        ),
    )
    p.add_argument("--model", default=None, help="覆盖 provider 默认 model")
    p.add_argument("--glossary", default=None, help="可选 glossary.json 路径")
    p.add_argument("--max-input-tokens", type=int, default=6000)
    p.add_argument("--max-cues-per-batch", type=int, default=40)
    p.add_argument("--context-prev", type=int, default=4)
    p.add_argument("--context-next", type=int, default=2)
    p.add_argument(
        "--emit-srt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="输出目标语言 SRT（默认 on）",
    )
    p.add_argument(
        "--emit-vtt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="输出目标语言 VTT（默认 on）",
    )
    p.add_argument(
        "--speaker-prefix",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "SRT/VTT 每条 cue 是否加 'Speaker X: ' 前缀。"
            "auto = 仅在 ≥2 个非空 speaker 时加（默认，单人讲座不再被强加 'Speaker A:'）；"
            "always = 旧行为；never = 永不加。"
        ),
    )
    p.add_argument(
        "--render-only",
        action="store_true",
        dest="render_only",
        help=(
            "跳过 LLM / cache，仅根据现有 subtitles.<lang>.json 重渲染 SRT/VTT。"
            "用于在不重新调 LLM 的情况下切换 --speaker-prefix。与 --force* 互斥。"
        ),
    )
    # --force 三档（语义见 TranslateRequest.force_level）
    p.add_argument(
        "--force",
        action="store_true",
        help="覆盖 draft 的 subtitles.<lang>.json 并清空 work/translate.<lang>/；遇 reviewed/final 仍会拒绝",
    )
    p.add_argument(
        "--force-reviewed",
        action="store_true",
        help="允许覆盖 reviewed 状态（隐含 --force）",
    )
    p.add_argument(
        "--force-final",
        action="store_true",
        help="允许覆盖 final 状态（隐含 --force-reviewed）。**销毁人工 lock 产物，慎用**",
    )
    p.add_argument("--json-events", action="store_true", help="stderr NDJSON 事件协议")
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "只做 batch 切分 + token / cost 估算，不调 LLM、不动 workdir。"
            "优先级高于 --render-only / --force*（dry-run 是只读操作）。"
        ),
    )
    p.add_argument("--timeout", type=float, default=60.0)


def _resolve_force_level(args: argparse.Namespace) -> str | None:
    if getattr(args, "force_final", False):
        return "final"
    if getattr(args, "force_reviewed", False):
        return "reviewed"
    if args.force:
        return "draft"
    return None


def _print_dry_run(summary: dict[str, Any]) -> None:
    """``--dry-run`` 路径输出（stderr）。无 next-step 导览（还没真跑）。"""
    provider = summary.get("provider", "?")
    model = summary.get("model", "?")
    pt = int(summary.get("estPromptTokens", 0))
    ct = int(summary.get("estCompletionTokens", 0))
    cost = summary.get("estCostUsd")
    sys.stderr.write(
        "dry-run estimate:\n"
        f"  batches: {summary.get('batchCount', 0)}, "
        f"cues: {summary.get('cueCount', 0)}\n"
        f"  prompt tokens (est): ~{pt}\n"
        f"  completion tokens (est): ~{ct}\n"
        f"  est cost: {format_cost(cost)} ({provider}/{model})\n"
    )


def run(args: argparse.Namespace) -> int:
    req = TranslateRequest(
        workdir=Path(args.workdir),
        target_language=args.target_language,
        source_language=args.source_language,
        style=args.style,
        length_policy=args.length_policy,
        provider=args.provider,
        model=args.model,
        glossary_path=Path(args.glossary) if args.glossary else None,
        max_input_tokens=args.max_input_tokens,
        max_cues_per_batch=args.max_cues_per_batch,
        context_prev=args.context_prev,
        context_next=args.context_next,
        emit_srt=args.emit_srt,
        emit_vtt=args.emit_vtt,
        speaker_prefix=args.speaker_prefix,
        render_only=args.render_only,
        force_level=_resolve_force_level(args),
        json_events=args.json_events,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
    )
    try:
        summary: dict[str, Any] = run_translate(req)
    except FileNotFoundError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except FileExistsError as e:
        sys.stderr.write(f"error: {e}\n")
        return 3
    except LLMError as e:
        sys.stderr.write(f"LLM error: {e}\n")
        return 4

    # dry-run：单独 summary
    if summary.get("dryRun"):
        if not args.json_events:
            _print_dry_run(summary)
        return 0

    if not args.json_events:
        if summary.get("renderOnly"):
            sys.stderr.write(
                f"translate {args.target_language} re-render done: "
                f"{summary.get('cueCount', 0)} cues, "
                f"speaker-prefix={summary.get('speakerPrefix')!r} (no LLM)\n"
            )
        else:
            provider = summary.get("provider", args.provider)
            model = summary.get("model", "?")
            pt = int(summary.get("promptTokens", 0))
            ct = int(summary.get("completionTokens", 0))
            cost = estimate_cost(provider, model, pt, ct)
            rates = lookup_rates(provider, model)
            sys.stderr.write(
                f"translate {args.target_language} done: "
                f"overChar={summary.get('overCharLimitRate', 0):.0%}, "
                f"glossaryMiss={summary.get('glossaryMissRate', 0):.0%}\n"
                f"  tokens: prompt={pt}, completion={ct} (total={pt + ct})\n"
            )
            if rates is not None:
                sys.stderr.write(
                    f"  est cost: {format_cost(cost)} ({provider}/{model} "
                    f"@ ${rates[0]:.2f} + ${rates[1]:.2f} per M)\n"
                )
            else:
                sys.stderr.write(
                    f"  est cost: (unknown rate for {provider}/{model})\n"
                )
            wd = args.workdir
            tgt = args.target_language
            sys.stderr.write(
                "next steps:\n"
                f"  voxkit quality {wd}                                       # 质量报告\n"
                f"  voxkit translate {wd} --target-language {tgt} --render-only --speaker-prefix never  # 改前缀不重 LLM\n"
                f"  voxkit review confirm {wd} --target {tgt}                # 人工 confirm → 锁 reviewed\n"
            )
    return 0
