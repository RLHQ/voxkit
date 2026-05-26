"""``voxkit proofread`` 子命令入口。

只负责 argparse → ``ProofreadRequest`` 的转换，业务在
:mod:`voxkit.core.proofread_pipeline` 里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from voxkit.core.pricing import format_cost, lookup_rates
from voxkit.core.proofread_pipeline import ProofreadRequest, run_proofread
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
        "    final             --force-final                覆盖 final（销毁人工 lock）\n"
        "改 glossary 后 cache 会因 glossaryHash 变化而失效，仍需对应 --force-*\n"
        "通过 gate（artifact 文件本身存在）。\n"
    )
    p = sub.add_parser(
        "proofread",
        help=(
            "对 subtitles.cues.json 跑 LLM 校对，产出 subtitles.proofread.json "
            "(state=draft)。需要环境变量 DEEPSEEK_API_KEY 或对应 provider 的 key"
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("workdir", help="voxkit transcribe 的 workdir（含 subtitles.cues.json）")
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
    p.add_argument(
        "--language",
        default=None,
        help="校对语言代码；缺省继承 cues.params.language，再缺省即 auto",
    )
    p.add_argument(
        "--edit-level",
        choices=["punctuation", "light", "standard", "strict"],
        default="standard",
        help="编辑强度（默认 standard）",
    )
    p.add_argument("--glossary", default=None, help="可选 glossary.json 路径")
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=6000,
        help="单 batch 输入 token 上限（默认 6000；保守估算 CJK 0.5、Latin 0.25 token/char）",
    )
    p.add_argument(
        "--max-cues-per-batch",
        type=int,
        default=40,
        help="单 batch cue 数上限（默认 40）",
    )
    p.add_argument("--context-prev", type=int, default=8, help="batch 上文 cue 数（默认 8）")
    p.add_argument("--context-next", type=int, default=4, help="batch 下文 cue 数（默认 4）")
    # --force 三档（语义见 ProofreadRequest.force_level）：
    #   --force          → 只覆盖 draft（默认安全档）
    #   --force-reviewed → 也覆盖 reviewed（确认要丢人工 confirm 的产物）
    #   --force-final    → 也覆盖 final（确认要丢人工锁定的产物）
    p.add_argument(
        "--force",
        action="store_true",
        help="覆盖 draft 状态的 subtitles.proofread.json 并清空 work/proofread/；遇 reviewed/final 仍会拒绝",
    )
    p.add_argument(
        "--force-reviewed",
        action="store_true",
        help="允许覆盖 reviewed 状态（隐含 --force）。注意会丢失人工 confirm 元数据",
    )
    p.add_argument(
        "--force-final",
        action="store_true",
        help="允许覆盖 final 状态（隐含 --force-reviewed）。**销毁人工 lock 产物，慎用**",
    )
    p.add_argument(
        "--json-events",
        action="store_true",
        help="stderr 改为 NDJSON 事件协议（机器消费）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "只做 batch 切分 + token / cost 估算，不调 LLM、不动 workdir。"
            "用来在花钱前先看一眼这批要烧多少美刀。"
            "传入时 --force* 一律忽略（dry-run 是只读操作）。"
        ),
    )
    p.add_argument("--timeout", type=float, default=60.0, help="单次 LLM 请求超时（秒）")


def _resolve_force_level(args: argparse.Namespace) -> str | None:
    """三档 force 旗标 → ``ForceLevel`` 字符串。任一高级位隐含低级。"""
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
    req = ProofreadRequest(
        workdir=Path(args.workdir),
        provider=args.provider,
        model=args.model,
        language=args.language,
        edit_level=args.edit_level,
        glossary_path=Path(args.glossary) if args.glossary else None,
        max_input_tokens=args.max_input_tokens,
        max_cues_per_batch=args.max_cues_per_batch,
        context_prev=args.context_prev,
        context_next=args.context_next,
        force_level=_resolve_force_level(args),
        json_events=args.json_events,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
    )
    try:
        summary: dict[str, Any] = run_proofread(req)
    except FileNotFoundError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except FileExistsError as e:
        sys.stderr.write(f"error: {e}\n")
        return 3
    except LLMError as e:
        sys.stderr.write(f"LLM error: {e}\n")
        return 4

    # dry-run：单独 summary，没必要打 next steps（还没真跑过）
    if summary.get("dryRun"):
        if not args.json_events:
            _print_dry_run(summary)
        return 0

    if not args.json_events:
        provider = summary.get("provider", args.provider)
        model = summary.get("model", "?")
        pt = int(summary.get("promptTokens", 0))
        ct = int(summary.get("completionTokens", 0))
        cost = None
        # 价格查表用 manifest 实际记录的 model（与正式 LLM 调用一致），
        # 而非 args.model（args.model 为 None 时是 provider 默认值）。
        from voxkit.core.pricing import estimate_cost as _ec
        cost = _ec(provider, model, pt, ct)
        rates = lookup_rates(provider, model)
        sys.stderr.write(
            f"proofread done: cues={summary.get('changedCueRate', 0):.0%} changed, "
            f"{summary.get('reviewCueRate', 0):.0%} need review\n"
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
        sys.stderr.write(
            "next steps:\n"
            f"  voxkit reseg {wd}                          # 双 pass：用 corrected 标点再切（推荐）\n"
            f"  voxkit translate {wd} --target-language zh # 翻译\n"
            f"  voxkit quality {wd}                        # 质量报告\n"
            f"  voxkit review confirm {wd}                 # 人工 confirm → 锁 reviewed\n"
        )
    return 0
