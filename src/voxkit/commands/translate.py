"""``voxkit translate`` 子命令入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from voxkit.core.translate_pipeline import TranslateRequest, run_translate
from voxkit.llm.errors import LLMError


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "translate",
        help=(
            "把字幕翻译到目标语言。输入优先 subtitles.proofread.json，缺失回落 "
            "subtitles.cues.json。需要 DEEPSEEK_API_KEY 或对应 provider key"
        ),
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
    p.add_argument("--provider", default="deepseek", help="LLM provider 名（默认 deepseek）")
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
    p.add_argument("--timeout", type=float, default=60.0)


def _resolve_force_level(args: argparse.Namespace) -> str | None:
    if getattr(args, "force_final", False):
        return "final"
    if getattr(args, "force_reviewed", False):
        return "reviewed"
    if args.force:
        return "draft"
    return None


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

    if not args.json_events:
        sys.stderr.write(
            f"translate {args.target_language} done: "
            f"overChar={summary.get('overCharLimitRate', 0):.0%}, "
            f"glossaryMiss={summary.get('glossaryMissRate', 0):.0%}, "
            f"{summary.get('promptTokens', 0)} + {summary.get('completionTokens', 0)} tokens\n"
        )
    return 0
